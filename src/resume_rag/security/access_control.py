from typing import List, Dict, Any, Optional, Set
from resume_rag.domain.models import User, Role, Permission
import logging

logger = logging.getLogger(__name__)

class AccessControl:
    def __init__(self, config_manager):
        self.config = config_manager
        access_config = config_manager.get_access_control_config()

        # Role-based permissions
        self.role_permissions = {
            Role.ADMIN: {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ANALYZE},
            Role.HR_MANAGER: {Permission.READ, Permission.ANALYZE},
            Role.RECRUITER: {Permission.READ},
            Role.ANALYST: {Permission.READ, Permission.ANALYZE}
        }

        self.department_categories = access_config.department_categories

    @staticmethod
    def validate_user(user_data: Dict[str, Any]) -> User:
        # Create a user
        return User(**user_data)

    def check_permission(self, user: User, permission: Permission) -> bool:
        # If a user has the permissions
        if not isinstance(user, User):
            logger.error("Invalid user object")
            return False

        role_key = Role(user.role) if isinstance(user.role, str) else user.role
        user_permissions = self.role_permissions.get(role_key, set())
        has_permission = permission in user_permissions

        if not has_permission:
            logger.warning(f"User {user.user_id} with role {user.role} denied {permission} permission")

        return has_permission

    def get_allowed_categories(self, user: User) -> Optional[Set[str]]:
        if not isinstance(user, User):
            logger.error("Invalid user object")
            return None

        # Admin can see all categories
        if user.role == Role.ADMIN:
            return None # all categories

        # Use explicitly set allowed categories if available
        if user.allowed_categories:
            return user.allowed_categories

        # Use department-based categories
        if user.department and user.department in self.department_categories:
            return set(self.department_categories[user.department])

        # Default: No specific restrictions
        return None

    def create_filter(self, user: User) -> Optional[Dict[str, Any]]:
        if not isinstance(user, User):
            logger.error("Invalid user object")
            return None

        allowed_categories = self.get_allowed_categories(user)

        if allowed_categories is None:
            return None  # No filtering needed

        return {"category": {"$in": list(allowed_categories)}}

    def filter_results(self, user: User, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Filter the results based on user access rights
        if not isinstance(user, User):
            logger.error("Invalid user object")
            return []

        if not results:
            return results

        # Admin sees everything
        if user.role == Role.ADMIN:
            return results

        allowed_categories = self.get_allowed_categories(user)

        # If no category restrictions, return all results
        if allowed_categories is None:
            return results

        # Filter results by allowed categories
        filtered_results = []
        for result in results:
            document = result.get("document") if isinstance(result, dict) else None
            metadata = getattr(document, "metadata", None)

            if isinstance(metadata, dict):
                doc_category = metadata.get("category", "Unknown")
            else:
                doc_category = getattr(metadata, "category", "Unknown")

            if doc_category in allowed_categories:
                filtered_results.append(result)
            else:
                logger.debug(f"Filtered out document with category {doc_category} for user {user.user_id}")

        logger.info(f"Filtered {len(results)} -> {len(filtered_results)} results for user {user.user_id}")
        return filtered_results

    def log_access(self, user: User, action: str, resource: str, success: bool):
        """Log access attempts for audit purposes."""
        if not isinstance(user, User):
            logger.error("Cannot log access for invalid user")
            return

        status = "SUCCESS" if success else "DENIED"
        logger.info(f"ACCESS_LOG: User={user.user_id}, Role={user.role}, "
                    f"Action={action}, Resource={resource}, Status={status}")

    def get_user_permissions(self, user: User) -> Set[Permission]:
        if not isinstance(user, User):
            return set()

        role_key = Role(user.role) if isinstance(user.role, str) else user.role
        return self.role_permissions.get(role_key, set())

    def can_access_category(self, user: User, category: str) -> bool:
        if not isinstance(user, User) or not category:
            return False

        allowed_categories = self.get_allowed_categories(user)

        # If no restrictions, allow access
        if allowed_categories is None:
            return True

        return category in allowed_categories